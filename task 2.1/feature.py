

class feature:

    def __init__(self, data):
        self.c = data.pop(-1)
        self.coordinates = data

    def get_vote(self):
        return self.c

    def get_coords(self):
        return self.coordinates