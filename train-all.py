import train

from pathlib import *
from shutil  import *


def main():
    for i in range(7, 10):
        train.main()

        src = Path('./model/cost.h5')
        to  = Path(f'./model/cost-1024x4-1000x1000x{i + 1:02d}00.h5')

        copy(str(src), str(to))


if __name__ == '__main__':
    main()
