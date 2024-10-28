class Item(object):
    """Lớp đại diện cho một mục có thể được thêm vào container của một vấn đề"""

    def __init__(self, shape, weight, value):
        self.shape = shape
        self.weight = weight
        self.value = value
        
    