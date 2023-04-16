class forecasterType:
    Original = 0 # Ashesh version
    PONI = 1

    @classmethod
    def name(cls, enum_type):
        for key, value in cls.__dict__.items():
            if enum_type == value:
                return key

    @classmethod
    def from_name(cls, enum_type_str):
        for key, value in cls.__dict__.items():
            if key == enum_type_str:
                return value
        assert f'{cls.__name__}:{enum_type_str} doesnot exist.'