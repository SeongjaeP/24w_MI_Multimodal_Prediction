import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers import convert_color_space

file_path = 'C:\\Users\\Administrator\\24w_MI_Multimodal_Prediction\\data\\echo\\94106955_0001.dcm'
#file_path = 'C:\\Users\\Administrator\\24w_MI_Multimodal_Prediction\\data\\echo\\94106955_0050.dcm'
# read in the DICOM with the pydicom module
dicom_data = pydicom.dcmread(file_path)

# print the DICOM metadata
for element in dicom_data:
    print(element)

# note the value for Photometric Interpretation that was printed, it should show:
# (0028, 0004) Photometric Interpretation          CS: 'YBR_FULL_422'
# we need to convert from YBR_FULL_422 to RGB to display the image properly
images_rgb = convert_color_space(dicom_data.pixel_array, "YBR_FULL_422", "RGB", per_frame=True)
# plot the first frame/image
plt.imshow(images_rgb[0])
plt.show()





# import matplotlib.pyplot as plt
# import pydicom
# from pydicom.pixel_data_handlers.util import apply_modality_lut

# file_path = 'C:\\Users\\Administrator\\24w_MI_Multimodal_Prediction\\data\\echo\\94106955_0050.dcm'
# dicom_data = pydicom.dcmread(file_path)

# # 픽셀 데이터가 있는지 확인
# if 'PixelData' in dicom_data:
#     # Modality LUT 적용 (일반적으로 이미지 밝기 조정에 사용됩니다)
#     image_data = apply_modality_lut(dicom_data.pixel_array, dicom_data)
    
#     # 멀티프레임 처리
#     num_frames = dicom_data.NumberOfFrames if 'NumberOfFrames' in dicom_data else 1
#     plt.figure(figsize=(10, 10))

#     # 추출하고 싶은 프레임의 수를 설정하세요. 여기서는 첫 4개의 프레임만 추출합니다.
#     for i in range(min(num_frames, 4)):  # 최대 4개의 프레임만 표시
#         frame = image_data[:, :, i] if num_frames > 1 else image_data
#         plt.subplot(2, 2, i + 1)
#         plt.imshow(frame, cmap='gray')
#         plt.title(f'Frame {i}')
#     plt.tight_layout()
#     plt.show()
# else:
#     print("이 DICOM 파일에는 픽셀 데이터가 없습니다.")
