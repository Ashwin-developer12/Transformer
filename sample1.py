class Demo:
    def __init__(self, num):
        self._num = num
        self.val = 0

    def disp(self):
        total = self._num + self.val
        return total
    
    def set_num(self, num):
        self._num = num

obj = Demo(70)
obj.val = 20
print(obj.disp())
obj.set_num(obj.val)
print(obj.disp())