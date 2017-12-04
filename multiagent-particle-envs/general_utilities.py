import errno
import os
def ensure_directory_exists(base_directory):
    """
    Makes a directory if it does not exist
    """
    try:
        os.makedirs(base_directory)
    except OSError as ex:
        if ex.errno != errno.EEXIST:
            raise ex

def load_dqn_weights_if_exist(dqns, weights_filename_prefix, weights_filename_extension=".h5"):
    """
    Loads weights if they exist, otherwise does nothing
    """
    for i, dqn in enumerate(dqns):
        # TODO should not work if only some weights available?
        dqn_filename = weights_filename_prefix + str(i) + weights_filename_extension
        if os.path.isfile(dqn_filename):
            print("Found old weights to use for agent {}".format(i))
            dqn.load(dqn_filename)

def save_dqn_weights(dqns, weights_filename_prefix, weights_filename_extension=".h5"):
    """
    Saves weights
    """
    ensure_directory_exists(os.path.splitext(weights_filename_prefix)[0])
    for i, dqn in enumerate(dqns):
        dqn_filename = weights_filename_prefix + str(i) + weights_filename_extension
        dqn.save(dqn_filename)

