from trainer.engine import main_worker
from trainer.utils import arg_parse


if __name__ == '__main__':
    args = arg_parse().parse_args()
    print(args)
    main_worker(args)


