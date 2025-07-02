# import os
# import cv2
# import numpy as np
# from tqdm import tqdm
# import easyocr

# def refine_flip_orientation(image, reader):
#     """
#     이미지에서 회전은 그대로 두고, 상하/좌우 반전만 시도해서 최적 OCR 방향 선택
#     """
#     candidates = {
#         'original': image,
#         'flip_ud': cv2.flip(image, 0),  # 상하 반전
#         'flip_lr': cv2.flip(image, 1),  # 좌우 반전
#         'flip_ud_lr': cv2.flip(cv2.flip(image, 0), 1),  # 상하 + 좌우
#     }

#     best_img = image
#     best_score = -1

#     for key, candidate in candidates.items():
#         try:
#             results = reader.readtext(candidate, detail=1)
#             if not results:
#                 continue

#             text_len = sum([len(res[1]) for res in results])
#             conf_avg = np.mean([res[2] for res in results])
#             score = text_len * conf_avg

#             if score > best_score:
#                 best_score = score
#                 best_img = candidate

#         except Exception as e:
#             print(f"OCR error on {key}: {e}")
#             continue

#     return best_img


# if __name__ == '__main__':
#     print("Initializing EasyOCR Reader...")
#     reader = easyocr.Reader(['ko', 'en'], gpu=True)
#     print("EasyOCR initialized.")

#     # 기존 1차 교정 이미지 폴더
#     # src_dir = './data/processed/06_ocr_then_line/'  # ← 49개 성공한 폴더로 바꾸기
#     # dst_dir = './data/processed/07_ocr_flip_refined/'
#     src_dir = './data/processed/02_debug_ocr_then_line_49/80/'
#     dst_dir = './data/processed/02-1_ocr_flip_refined/'
#     os.makedirs(dst_dir, exist_ok=True)

#     image_files = [f for f in os.listdir(src_dir) if not f.startswith('.')]

#     for file_name in tqdm(image_files, desc="Refining with flip correction"):
#         src_path = os.path.join(src_dir, file_name)
#         img = cv2.imread(src_path)
#         if img is None: continue

#         refined = refine_flip_orientation(img, reader)

#         dst_path = os.path.join(dst_dir, file_name)
#         cv2.imwrite(dst_path, refined)

#     print(f"\n✅ Done! Results saved in {dst_dir}")

import os
import cv2
import numpy as np
from tqdm import tqdm
import easyocr
import torch  # 추가

def refine_flip_orientation(image, reader):
    candidates = {
        'original': image,
        'flip_ud': cv2.flip(image, 0),
        'flip_lr': cv2.flip(image, 1),
        'flip_ud_lr': cv2.flip(cv2.flip(image, 0), 1),
    }

    best_img = image
    best_score = -1

    for key, candidate in candidates.items():
        try:
            results = reader.readtext(candidate, detail=1)
            if not results:
                continue

            text_len = sum([len(res[1]) for res in results])
            conf_avg = np.mean([res[2] for res in results])
            score = text_len * conf_avg

            if score > best_score:
                best_score = score
                best_img = candidate

        except Exception as e:
            print(f"OCR error on {key}: {e}")
            continue

    return best_img


if __name__ == '__main__':
    # 디바이스 확인 (PyTorch 기반)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # EasyOCR 초기화
    print("Initializing EasyOCR Reader...")
    reader = easyocr.Reader(['ko', 'en'], gpu=(device.type == 'cuda'))
    print("EasyOCR initialized.")

    # 경로 설정
    src_dir = './data/processed/02_debug_ocr_then_line_49/80/'
    dst_dir = './data/processed/02-1_ocr_flip_refined/'
    os.makedirs(dst_dir, exist_ok=True)

    image_files = [f for f in os.listdir(src_dir) if not f.startswith('.')]

    for file_name in tqdm(image_files, desc="Refining with flip correction"):
        src_path = os.path.join(src_dir, file_name)
        img = cv2.imread(src_path)
        if img is None: continue

        refined = refine_flip_orientation(img, reader)
        dst_path = os.path.join(dst_dir, file_name)
        cv2.imwrite(dst_path, refined)

    print(f"\n✅ Done! Results saved in {dst_dir}")