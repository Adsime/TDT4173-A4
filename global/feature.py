

class feature:

    def __init__(self, data, is_adaboost=False):
        self.index = int((data.pop(0) + 1 if is_adaboost else -1))
        self.target = int(data.pop(0 if is_adaboost else -1))
        self.coordinates = data

    def get_target(self):
        return self.target

    def get_coords(self):
        return self.coordinates

    def get_index(self):
        return self.index
