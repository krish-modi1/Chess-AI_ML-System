import torch
import torch.multiprocessing as mp
import time
import os
import numpy as np
import concurrent.futures
import queue

from game_engine.cnn import ChessCNN

class InferenceServer:
    def __init__(self, model_path, batch_size=512, timeout=0.1, streams=4,
                 shm_inp=None, shm_pol=None, shm_val=None):
        self.model_path = model_path
        self.batch_size = batch_size
        self.timeout = timeout
        self.num_streams = streams

        # Shared-memory transport: bulk leaf/result tensors live in these per-worker-slotted
        # buffers (NUM_WORKERS, WORKER_BATCH_SIZE, ...). When present, the queues carry only
        # tiny (worker_id, N) control signals instead of pickled tensors. None → legacy path.
        self.shm_inp = shm_inp
        self.shm_pol = shm_pol
        self.shm_val = shm_val
        self.shm = shm_inp is not None

        self.input_queue = mp.Queue()
        self.output_queues = {}

    def register_worker(self, worker_id):
        self.output_queues[worker_id] = mp.Queue()
        return self.output_queues[worker_id]

    def process_batch(self, batch_data, stream_idx, model, device):
        if not batch_data: return
        stream = self.streams[stream_idx]

        # ── Shared-memory path ───────────────────────────────────────────────────
        # batch_data is a list of (worker_id, N). Gather each worker's N leaves from its
        # shared input slot into this stream's pinned staging buffer (one contiguous H2D),
        # run the model, then scatter raw logits back into the shared output slots and
        # signal each worker with a tiny (N,) tuple. Each worker_id appears at most once
        # per in-flight batch (single-outstanding-request invariant), and staging is
        # per-stream, so concurrent process_batch calls never touch the same memory.
        if self.shm:
            stage = self.stages[stream_idx]
            cursor = 0
            for wid, n in batch_data:
                stage[cursor:cursor + n] = self.shm_inp[wid, :n]
                cursor += n
            total = cursor

            with torch.cuda.stream(stream):
                mega_gpu = stage[:total].to(device, non_blocking=True)
                with torch.no_grad():
                    pol_gpu, val_gpu = model(mega_gpu)
                del mega_gpu
            # `stream` is a NON-default stream; the model ran on it. `.cpu()` below syncs the
            # DEFAULT stream, NOT `stream`, so without this it can copy BEFORE the compute finishes
            # → ~5% non-deterministic / wrong leaf evals (verified: self-inconsistent reads). Wait
            # for the producing stream first.
            stream.synchronize()
            pol = pol_gpu.cpu()
            val = val_gpu.cpu()
            del pol_gpu, val_gpu

            cursor = 0
            for wid, n in batch_data:
                self.shm_pol[wid, :n] = pol[cursor:cursor + n]
                self.shm_val[wid, :n] = val[cursor:cursor + n]
                cursor += n
                self.output_queues[wid].put((n,))
            return

        # ── Legacy path (tensors pickled through the queue) ──────────────────────
        worker_ids = [item[0] for item in batch_data]
        raw_tensors = [item[1] for item in batch_data]

        sizes = [t.shape[0] if t.ndim == 4 else 1 for t in raw_tensors]

        processed_tensors = [t if t.ndim == 4 else t.unsqueeze(0) for t in raw_tensors]
        mega_batch = torch.cat(processed_tensors, dim=0)
        del processed_tensors

        with torch.cuda.stream(stream):
            mega_batch_gpu = mega_batch.to(device, dtype=torch.float32, non_blocking=True)
            del mega_batch
            with torch.no_grad():
                policies_gpu, values_gpu = model(mega_batch_gpu)
            del mega_batch_gpu

        stream.synchronize()   # same non-default-stream hazard as the SHM path: sync before .cpu()
        policies = policies_gpu.cpu().numpy()
        values = values_gpu.cpu().numpy()
        del policies_gpu, values_gpu

        cursor = 0
        for i, wid in enumerate(worker_ids):
            size = sizes[i]
            p_slice = policies[cursor : cursor + size]
            v_slice = values[cursor : cursor + size]
            cursor += size

            if size == 1 and raw_tensors[i].ndim == 3:
                self.output_queues[wid].put((p_slice[0], v_slice[0]))
            else:
                self.output_queues[wid].put((p_slice, v_slice))

    def loop(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ChessCNN().to(self.device)
        self.streams = [torch.cuda.Stream() for _ in range(self.num_streams)]
        self.current_stream_idx = 0

        # One pinned staging buffer per stream (so concurrent process_batch calls on
        # different streams never share a buffer). Sized batch_size + one worker slot to
        # cover the gather loop's last-item overshoot. Only needed for the SHM path.
        if self.shm:
            assert self.shm_inp.is_shared(), "shm_inp not in shared memory — check Process args"
            slot = self.shm_inp.shape[1]
            cap = self.batch_size + slot
            pin = (self.device.type == "cuda")   # pinning only helps (and only works) with CUDA
            self.stages = [torch.empty((cap, 120, 8, 8), dtype=torch.float32, pin_memory=pin)
                           for _ in range(self.num_streams)]
        
        print(f"Server: Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
        # strict=False: a pre-aux champion checkpoint lacks the aux-head keys; inference only
        # reads policy+value, so random-init aux heads are harmless. (See auxiliary-targets plan.)
        model.load_state_dict(state, strict=False)
        model.eval()
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_streams)
        print(f"Server Ready: Batch={self.batch_size}, Timeout={self.timeout}s, Streams={self.num_streams}, Device={self.device}")

        last_successful_batch_time = time.time()
        # Self-kill if the server processes NO batch for this long (a real hang). With heavy
        # worker oversubscription the server is busier, not idle, so this won't fire in normal
        # operation; env-tunable in case startup/wind-down windows ever need more slack.
        deadlock_timeout = int(os.environ.get("SERVER_DEADLOCK_TIMEOUT", 1800))
        pending_futures = []
        batches_since_cache_clear = 0

        while True:
            batch_data = []
            current_time = time.time()

            if current_time - last_successful_batch_time > deadlock_timeout:
                print(f"🚨 DEADLOCK DETECTED: No batch processed in {deadlock_timeout}s")
                return

            current_batch_count = 0
            start_time = time.time()

            while current_batch_count < self.batch_size:
                try:
                    item = self.input_queue.get(timeout=0.01)
                    if item == "STOP":
                        if batch_data:
                            executor.submit(self.process_batch, batch_data, self.current_stream_idx, model, self.device)
                        executor.shutdown(wait=True)
                        return

                    if self.shm:
                        # item is (worker_id, N) — N leaves already written to shm_inp[wid].
                        wid, item_size = item
                    else:
                        tensor = item[1]
                        item_size = tensor.shape[0] if tensor.ndim == 4 else 1
                    batch_data.append(item)
                    current_batch_count += item_size

                except queue.Empty:
                    pass

                if time.time() - start_time > self.timeout:
                    break

            if batch_data:
                stream_idx = self.current_stream_idx
                self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
                fut = executor.submit(self.process_batch, batch_data, stream_idx, model, self.device)
                pending_futures.append(fut)

                last_successful_batch_time = time.time()
                batches_since_cache_clear += 1

                # Collect done futures to free batch_data references
                pending_futures = [f for f in pending_futures if not f.done()]

                # Periodically free CUDA allocator cache
                if batches_since_cache_clear >= 200:
                    torch.cuda.empty_cache()
                    batches_since_cache_clear = 0