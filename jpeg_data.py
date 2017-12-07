import sys
import binascii

class JPEG(object):
    def __init__(self, path):
        self.heximg = self.read_jpeg(path)
        self.dct = []
    
    def read_jpeg(self, path):
        with open(path, "rb") as f:
            bstr = f.read()
            bimg = binascii.hexlify(bstr)
        return [bimg[i:i+2].decode("utf-8").upper() for i in range(0, len(bimg), 2)]

    def parse(self):
        # start jpeg data
        # SOI
        soi, self.heximg = self.heximg[0:2], self.heximg[2:]
        print("====== SOI =====")
        print(soi)

        # get Application Type
        # Now I suppose that read jpeg use JFJF, not Exif
        # In the future, I am going to accept Exif
        print("===== AppData =====")
        marker_recog, marker, self.heximg = self.heximg[0], self.heximg[1], self.heximg[2:]
        print("marker: ", marker)
        length = eval("0x"+self.heximg[0]+self.heximg[1])
        self.heximg = self.heximg[length:]
        print("length: ", length)

        # get DQT segmentation
        print("===== DCT Segmentation =====")
        while True:
            if self.heximg[0] != "FF" or self.heximg[1] != "DB":
                break
            
            # DQT information
            self.heximg = self.heximg[2:]
            length = eval("0x"+self.heximg[0]+self.heximg[1])
            table_id = eval("0x" + self.heximg[2])
            print("length: ", length)
            print("table_id: ", table_id)

            # get DQT tables
            dct_table = []
            for i in range(3, length):
                dct_table.append(self.heximg[i])
            self.dct.append(dct_table)
            self.heximg = self.heximg[length:]
            print("==========")


if __name__ == "__main__":
    jpeg_data = JPEG(sys.argv[1])
    jpeg_data.parse()
