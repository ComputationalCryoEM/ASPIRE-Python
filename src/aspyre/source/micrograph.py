from aspyre.source import ImageSource


class Micrograph(ImageSource):
    """
    A Micrograph is a more general type of (the existing) Starfile class, in which there are no global
    parameters applicable to all micrographs.
    """

    @classmethod
    def from_image(cls, image):
        """
        Construct from an already constructed Image object
        Useful if we just want to utilize this class for serialization purposes.
        :param image: and Image object
        :return: and initialized Micrograph object
        """
        pass

    def save(self, folder):
        """
        Save micrograph files to folder
        :param folder:
        :return:
        """
        pass
