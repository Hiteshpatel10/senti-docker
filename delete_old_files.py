import time
import os
import datetime
import daemon
import schedule

def delete_old_files():
    directory = '/path/to/your/directory'

    # get the current time
    current_time = datetime.datetime.now()

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # check if the file is a file (not a directory) and is older than an hour
        if os.path.isfile(filepath) and current_time - datetime.datetime.fromtimestamp(os.path.getctime(filepath)) > datetime.timedelta(hours=1):
            os.remove(filepath)

def main():
    # run the job every hour
    schedule.every().hour.do(delete_old_files)

    while True:
        schedule.run_pending()
        time.sleep(600)

if __name__ == '__main__':
    with daemon.DaemonContext():
        main()
