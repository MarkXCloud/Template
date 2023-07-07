from template import val_args, launch_val

def main():
    parser = val_args()
    args = parser.parse_args()
    launch_val(args)

if __name__ == '__main__':
    main()
