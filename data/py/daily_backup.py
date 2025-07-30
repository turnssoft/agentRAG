# daily_backup.py
# Author: Brenda Marketing
# Date: 2025-07-20
# Description: A simple script to back up daily sales logs to a compressed archive.

import os
import shutil
from datetime import datetime

# --- Configuration ---
# Define the source directory where daily transaction logs are stored.
SOURCE_DIRECTORY = "/var/log/pos/daily_sales"

# Define the destination directory for the backups.
# In a real scenario, this would likely be a mounted network drive.
BACKUP_DESTINATION = "/mnt/network_storage/backups/sales_archive"

def create_backup_archive():
    """
    Finds today's sales data and compresses it into a timestamped zip file.
    """
    # 1. Get today's date to identify the correct folder and name the archive.
    today = datetime.now()
    date_str = today.strftime("%Y-%m-%d")
    
    # The source folder is expected to be named with the current date.
    source_path = os.path.join(SOURCE_DIRECTORY, date_str)
    
    # 2. Check if the source directory for today actually exists.
    if not os.path.isdir(source_path):
        print(f"ERROR: Source directory not found for today: {source_path}")
        print("Backup operation cancelled. No sales data to archive for this date.")
        return

    # 3. Define the name of the output archive file.
    archive_name = f"sales_backup_{date_str}"
    
    # The full path for the new zip file (without the .zip extension yet).
    output_path = os.path.join(BACKUP_DESTINATION, archive_name)

    print(f"INFO: Starting backup of '{source_path}' to '{output_path}.zip'")

    try:
        # 4. Create the compressed archive.
        # The 'make_archive' function is powerful. It takes the desired output path (without extension),
        # the format ('zip' in this case), and the directory to archive.
        shutil.make_archive(output_path, 'zip', source_path)
        
        print(f"SUCCESS: Backup completed successfully.")
        print(f"Archive created at: {output_path}.zip")
    except FileNotFoundError:
        print(f"ERROR: The backup destination directory does not exist: {BACKUP_DESTINATION}")
        print("Please ensure the network drive is mounted and accessible.")
    except Exception as e:
        print(f"FATAL: An unexpected error occurred during the backup process: {e}")

if __name__ == "__main__":
    print("--- Pizza Boys Daily Backup Utility ---")
    
    # Ensure the backup destination exists before starting.
    if not os.path.exists(BACKUP_DESTINATION):
        print(f"WARNING: Backup destination '{BACKUP_DESTINATION}' not found. Attempting to create it.")
        try:
            os.makedirs(BACKUP_DESTINATION)
            print(f"INFO: Successfully created backup directory.")
        except Exception as e:
            print(f"FATAL: Could not create backup directory. Reason: {e}")
            # Exit if we can't create the destination, as the backup will fail.
            exit(1)

    create_backup_archive()
    
    print("--- Backup script finished. ---")
