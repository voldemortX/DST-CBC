import argparse
from torchvision.datasets.utils import download_and_extract_archive


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser(description='Downloader')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Cifar10 (default: cifar10)')
    parser.add_argument('--base', type=str, default='../',
                        help='Dataset directory (default: ../)')
    args = parser.parse_args()
    if args.dataset == 'cifar10':
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    else:
        raise NotImplementedError

    download_and_extract_archive(url, args.base, filename=filename, md5=tgz_md5)
