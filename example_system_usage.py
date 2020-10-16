from tagging_system import Document, SegmentationModel, FindingFormulasModel


seg_model = SegmentationModel(
    path_to_model = './models/MaskRCNN_Resnext101_32x8d_FPN_3X.pth',
    path_to_cfg_config = './configs/DLA_mask_rcnn_X_101_32x8d_FPN_3x.yaml',
    device = 'cpu',
    score_thresh_test = 0.5
)

find_model = FindingFormulasModel(
    path_to_model =  './models/AMATH512_e1GTDB.pth',
    score_thresh_test = 0.7
)

doc = Document(
        pdf_path = './uploaded_files/color_2.pdf', 
        segmentation_model = seg_model,
        finding_formulas_model = find_model, 
        layout_type = 1,
        document_type = 2,
        dpi = 900,
        langs = ['rus', 'eng'],
        tessdata_dir = '/usr/share/tesseract-ocr/4.00/tessdata'
    )

result = doc.convert(output_type='docx', output_filename='tagging_system_example.docx', to_zip = False)
print(result)
