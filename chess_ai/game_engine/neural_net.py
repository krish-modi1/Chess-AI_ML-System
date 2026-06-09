import torch
import torch.multiprocessing as mp
import time
import numpy as np
import concurrent.futures
import queue

from game_engine.cnn import ChessCNN

class InferenceServer:
    def __init__(self, model_path, batch_size=512, timeout=0.1, streams=4):
        self.model_path = model_path
        self.batch_size = batch_size
        self.timeout = timeout
        self.num_streams = streams
        
        self.input_queue = mp.Queue()
        self.output_queues = {} 

    def register_worker(self, worker_id):
        self.output_queues[worker_id] = mp.Queue()
        return self.output_queues[worker_id]

    def process_batch(self, batch_data, stream, model, device):
        if not batch_data: return

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

        # .cpu() is blocking — synchronizes the stream implicitly
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
        
        print(f"Server: Loading model from {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        if any(k.startswith('_orig_mod.') for k in state):
            state = {k.removeprefix('_orig_mod.'): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_streams)
        print(f"Server Ready: Batch={self.batch_size}, Timeout={self.timeout}s, Streams={self.num_streams}, Device={self.device}")

        last_successful_batch_time = time.time()
        deadlock_timeout = 1800
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
                            stream = self.streams[self.current_stream_idx]
                            executor.submit(self.process_batch, batch_data, stream, model, self.device)
                        executor.shutdown(wait=True)
                        return

                    tensor = item[1]
                    item_size = tensor.shape[0] if tensor.ndim == 4 else 1
                    batch_data.append(item)
                    current_batch_count += item_size

                except queue.Empty:
                    pass

                if time.time() - start_time > self.timeout:
                    break

            if batch_data:
                stream = self.streams[self.current_stream_idx]
                self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
                fut = executor.submit(self.process_batch, batch_data, stream, model, self.device)
                pending_futures.append(fut)

                last_successful_batch_time = time.time()
                batches_since_cache_clear += 1

                # Collect done futures to free batch_data references
                pending_futures = [f for f in pending_futures if not f.done()]

                # Periodically free CUDA allocator cache
                if batches_since_cache_clear >= 200:
                    torch.cuda.empty_cache()
                    batches_since_cache_clear = 0