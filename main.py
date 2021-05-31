from engine.trainer import main_worker
from engine.utils import arg_parse


if __name__ == '__main__':
    args = arg_parse().parse_args()
    print(args)
    main_worker(args)


