import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='TIFFReadDirectory')
import warnings
warnings.filterwarnings("ignore")

from sty import fg


def print_error_message(message="", terminate=False):
    print(fg(255, 10, 10) + message + fg.rs, sep='\n')
    if terminate:
        exit()

def print_info_message(message=""):
    print(fg(0, 255, 10) + message + fg.rs, sep='\n')
