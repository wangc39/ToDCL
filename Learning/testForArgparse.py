# test.py
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--test_action", action='store_true')
    # 如果python testForArgparse.py --test_action 那么args.test_action为False
    # 如果不指定的话 则为True
    parser.add_argument("--test_action", action='store_false')


    args = parser.parse_args()
    action_val = args.test_action

    print(action_val)
