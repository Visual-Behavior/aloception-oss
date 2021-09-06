class SequenceMixin(object):
    def __init__(self, sequence_size: int = 1, sequence_skip: int = 0, **kwargs):
        """Sequence Mixin
        Parameters
        ----------
        sequence_size: int
            Size of sequence to load
        sequence_skip: int
            Number of frame to skip between each element of the sequence
        """

        super(SequenceMixin, self).__init__(**kwargs)

        self.sequence_size = sequence_size
        self.sequence_skip = sequence_skip
