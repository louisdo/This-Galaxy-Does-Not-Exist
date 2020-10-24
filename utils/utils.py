import yaml


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def get_config_yaml(config_path: str) -> dict:
        """
        Get config from yaml file

        input: 
            + config_path: path to your config file
        output:
            + a hash table containing your configurations
        """
        with open(config_path, 'r') as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)
