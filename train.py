from template import basic_args, launch

def main():
    parser = basic_args()
    args = parser.parse_args()
    launch(args)

if __name__ == '__main__':
    main()
