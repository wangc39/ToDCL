from argparse import ArgumentParser
import torch

def parse_test_opt():
    parser = ArgumentParser()

    # parser.add_argument('--model_checkpoint', type=str, default="gpt2")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path to the folder with the results")
    parser.add_argument("--saving_dir", type=str, default="result/", help="Path to the folder for saving")
    parser.add_argument("--load_model_base_file", type=str, default="/data/wangcong/CL-dialogue/runs", help="Path to the folder with model")
    parser.add_argument("--remake_test_file", action='store_true', help="to remake the test files")


    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for test") # test dimension equal to 1
    parser.add_argument("--responses_generate_times", type=int, default=5, help="The number of generated response")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--sample_dataset_radio", type=float, default=1.0, help="To sample the dataset for quick training")

    # parser.add_argument("--dataset_list", type=str, default="Ed,Wow,Daily", help="Path for saving")
    parser.add_argument("--dataset_list", type=str, default="Convai2,Ed,Wow,Daily,Cornell", help="Path for saving")

    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--setting", type=str, default="single", help="Path for saving")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--test_every_step", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="lenght of the generation")
    parser.add_argument("--debug", action='store_true', help="continual baseline")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers")
    parser.add_argument("--bottleneck_size", type=int, default=100)
    parser.add_argument("--number_of_adpt", type=int, default=40, help="number of adapterss")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--percentage_LAM0L", type=float, default=0.2, help="LAMOL percentage of augmented data used")
    parser.add_argument("--reg", type=float, default=0.01, help="CL regularization term")
    parser.add_argument("--episodic_mem_size", type=int, default=100, help="number of batch/sample put in the episodic memory")
    parser.add_argument('--CL', type=str, default="MULTI")
    parser.add_argument('--seed', default=42, type=int)


    args = parser.parse_args()

    return args

def parse_train_opt():

    parser = ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, default="gpt2")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for test") # test dimension equal to 1
    parser.add_argument("--responses_generate_times", type=int, default=5, help="The number of generated response")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")
    parser.add_argument("--sample_dataset_radio", type=float, default=1.0, help="To sample the dataset for quick training")

    parser.add_argument("--dataset_list", type=str, default="Convai2,Ed,Wow,Daily,Cornell", help="Path for saving")
    # parser.add_argument("--dataset_list", type=str, default="Ed,Daily", help="Path for saving")

    # parser.add_argument("--dataset_list", type=str, default="Ed,Wow,Daily", help="Path for saving")

    parser.add_argument("--max_history", type=int, default=15, help="max number of turns in the dialogue")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--setting", type=str, default="single", help="Path for saving")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--test_every_step", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="lenght of the generation")
    parser.add_argument("--debug", action='store_true', help="continual baseline")
    parser.add_argument("--n_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers")

    parser.add_argument("--bottleneck_size", type=int, default=100)
    parser.add_argument("--number_of_adpt", type=int, default=40, help="number of adapterss")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--percentage_LAM0L", type=float, default=0.2, help="LAMOL percentage of augmented data used")
    parser.add_argument("--reg", type=float, default=0.01, help="CL regularization term")
    parser.add_argument("--episodic_mem_size", type=int, default=100, help="number of batch/sample put in the episodic memory")

    parser.add_argument('--CL', type=str, default="MULTI")
    parser.add_argument('--seed', default=42, type=int)


    hyperparams = parser.parse_args()

    return hyperparams