class Base_c:
    def __init__(self, **kwargs):
        self.att1 = kwargs['att1']
        self.att2 = kwargs['att2']
        self.att3 = kwargs['att3']
    
    def print(self):
        print('att1', self.att1)
        print('att2', self.att2)
        print('att3', self.att3)


class Child1(Base_c):
    def __init__(self, att4, **kwargs):
        super(Child1, self).__init__(**kwargs)
        self.att4 = att4


    @classmethod
    def from_base_c(cls, base_c_obj, att4):
        print(base_c_obj.__dict__)


base_c1 = Base_c(att1=1,att2=2,att3=3)
child1 = Child1.from_base_c(base_c1, 4)
