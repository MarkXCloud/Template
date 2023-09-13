import os
from accelerate.commands.launch import launch_command_parser, launch_command


def main():
    parser = launch_command_parser()
    args = parser.parse_args()

    if args.gpu_ids:
        args.num_processes = len(args.gpu_ids.split(','))

    args.module = True

    args.training_script_args.insert(0, args.training_script)
    args.training_script = "template.main"

    launch_command(args)
