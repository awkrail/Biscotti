import os

def mkdir(dirname):
    if not os.path.exists(dirname):
        print("make ", dirname, " directory")
        os.mkdir(dirname)

def make_directory():
    print("=== make directory for dataset ===")
    mkdir("resized_images/")
    mkdir("opt_images/")
    mkdir("qopt_images/")
    mkdir("csv/")
    mkdir("train/")
    print("===> Done!")

if __name__ == "__main__":
    make_directory()