import numpy as np
def defense(args, it) -> list:
    '''
        read the assigned weight from the file and return the weight list
    '''
    rb_path = args.rb_rootpth + f"/{it}.txt"
    try:
        with open(rb_path, 'rb') as f:
            rb_weights = np.loadtxt(f, dtype=int).tolist()
        print(f"Read robust weights from {rb_path}")
    except FileNotFoundError:
        print(f"Robust weights file {rb_path} not found. Using default weights.")
        rb_weights = [100] * args.num_users

    return rb_weights