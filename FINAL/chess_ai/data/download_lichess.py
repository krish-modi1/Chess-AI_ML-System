"""
Lichess Database Downloader

Downloads Lichess PGN files with accurate progress tracking.
Progress bar and ETA based on actual file size being downloaded,
not hardcoded estimates.
"""

import subprocess
import os
from pathlib import Path
import argparse
import sys
import time
import urllib.request
import urllib.error


def format_size(bytes_val):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def get_remote_file_size(url, timeout=10):
    """
    Get actual file size from remote server using HEAD request.
    
    Args:
        url: URL of file
        timeout: Request timeout in seconds
        
    Returns:
        File size in bytes, or None if unable to determine
    """
    try:
        req = urllib.request.Request(url, method='HEAD')
        req.add_header('User-Agent', 'Mozilla/5.0')
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            size = response.headers.get('Content-Length')
            if size:
                return int(size)
    except Exception:
        pass
    
    return None


def download_lichess_month(year, month, output_dir="lichess_data"):
    """
    Download a single month's database with accurate progress tracking.
    
    Args:
        year: Year to download
        month: Month to download
        output_dir: Output directory path
        
    Returns:
        Path to downloaded file, or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    filename = f"lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
    url = f"https://database.lichess.org/standard/{filename}"
    output_path = output_dir / filename
    
    # Check if file already exists
    if output_path.exists():
        size = output_path.stat().st_size
        print(f"\n Already exists: {format_size(size)}")
        return output_path
    
    print(f"\n{'-'*70}")
    print(f" Downloading: {filename}")
    print(f" URL: {url}")
    print(f"{'-'*70}")
    
    # Get actual remote file size
    print(" Checking remote file size...")
    remote_size = get_remote_file_size(url)
    
    if remote_size:
        print(f" Remote size: {format_size(remote_size)}")
    else:
        print(" Warning: Could not determine remote file size")
        remote_size = None
    
    try:
        start_time = time.time()
        last_size = 0
        last_time = start_time
        
        # Run curl for download
        cmd = [
            "curl",
            "-L",
            "-C", "-",
            "-s",
            "-o", str(output_path),
            url
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Monitor while downloading
        while process.poll() is None:
            if output_path.exists():
                current_size = output_path.stat().st_size
                current_time = time.time()
                elapsed = current_time - start_time
                
                if current_size > last_size:
                    speed = (current_size - last_size) / (current_time - last_time)
                    speed_mb_s = speed / (1024**2)
                    
                    # Use actual remote size if available
                    if remote_size:
                        total_estimate = remote_size
                        percent = min(100, (current_size / total_estimate) * 100)
                        remaining = total_estimate - current_size
                        
                        if speed > 0:
                            eta_secs = remaining / speed
                            eta_mins = int(eta_secs / 60)
                            eta_secs = int(eta_secs % 60)
                        else:
                            eta_mins = 0
                            eta_secs = 0
                    else:
                        percent = 0
                        eta_mins = 0
                        eta_secs = 0
                    
                    # Progress bar
                    bar_width = 40
                    filled = int(bar_width * percent / 100)
                    bar = chr(9608) * filled + chr(9617) * (bar_width - filled)
                    
                    if remote_size:
                        print(f"\r [{bar}] {percent:5.1f}% | {format_size(current_size):>8} | "
                              f"{speed_mb_s:4.1f} MB/s | ETA {eta_mins:02d}:{eta_secs:02d}",
                              end='', flush=True)
                    else:
                        print(f"\r Downloaded: {format_size(current_size)} | "
                              f"{speed_mb_s:4.1f} MB/s",
                              end='', flush=True)
                    
                    last_size = current_size
                    last_time = current_time
            
            time.sleep(0.5)
        
        # Wait for process to complete
        retcode = process.wait()
        
        if retcode == 0 and output_path.exists():
            elapsed = time.time() - start_time
            size = output_path.stat().st_size
            elapsed_min = int(elapsed / 60)
            elapsed_sec = int(elapsed % 60)
            print(f"\n Downloaded: {format_size(size)} in {elapsed_min}m {elapsed_sec}s")
            return output_path
        else:
            print(f"\n Download failed")
            if output_path.exists():
                output_path.unlink()
            return None
    
    except KeyboardInterrupt:
        print(f"\n Cancelled")
        if output_path.exists():
            output_path.unlink()
        return None
    
    except Exception as e:
        print(f"\n Error: {e}")
        if output_path.exists():
            output_path.unlink()
        return None


def main():
    """Main entry point for downloader"""
    parser = argparse.ArgumentParser(description='Download Lichess databases')
    parser.add_argument('--start-year', type=int, default=2024,
                        help='Starting year')
    parser.add_argument('--start-month', type=int, default=2,
                        help='Starting month')
    parser.add_argument('--num-months', type=int, default=6,
                        help='Number of months')
    parser.add_argument('--output-dir', type=str, default='data/lichess_raw',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Generate months list
    months = []
    year, month = args.start_year, args.start_month
    
    for _ in range(args.num_months):
        months.append((year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1
    
    # Header
    print("\n" + "="*70)
    print(" Lichess Database Downloader")
    print("="*70)
    print(f" Downloading {len(months)} months:")
    for y, m in months:
        print(f" - {y}-{m:02d}")
    print("="*70)
    
    # Download
    downloaded = []
    for year, month in months:
        result = download_lichess_month(year, month, args.output_dir)
        if result:
            downloaded.append(result)
    
    # Footer
    print("\n" + "="*70)
    print(" Complete")
    print("="*70)
    print(f" Downloaded: {len(downloaded)}/{len(months)}")
    
    if downloaded:
        total = sum(f.stat().st_size for f in downloaded)
        print(f" Total size: {format_size(total)}")
        print(f" Saved to: {args.output_dir}/")
        
        print(f"\n Next step:")
        print(f" python3 data/validate_lichess.py {args.output_dir}")
        print(f"\n Then:")
        print(f" python3 data/download_standard_classical.py \\")
        print(f"     --input-dir {args.output_dir}")
    
    print("="*70 + "\n")
    
    return 0 if downloaded else 1


if __name__ == "__main__":
    sys.exit(main())
