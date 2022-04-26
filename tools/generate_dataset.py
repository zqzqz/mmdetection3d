from data_converter.opv2v_converter import OPV2V2KITTI

if __name__ == "__main__":
    converter = OPV2V2KITTI("/z/qzzhang/mvp/data", "/z/qzzhang/mvp/third_party/mmdetection3d/data/opv2v", "train")
    converter.convert()
    converter = OPV2V2KITTI("/z/qzzhang/mvp/data", "/z/qzzhang/mvp/third_party/mmdetection3d/data/opv2v", "validate")
    converter.convert()
    converter = OPV2V2KITTI("/z/qzzhang/mvp/data", "/z/qzzhang/mvp/third_party/mmdetection3d/data/opv2v", "test")
    converter.convert()