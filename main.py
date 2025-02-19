import os
from pdf2image import convert_from_path
import cv2
import numpy as np

# Konfiguráció
input_folder = r"C:\location"
output_folder = r"C:\location"
fps = 30  # Frames per second
display_time_per_image_seconds = 5  # Minden érvényes kép 5 másodpercig látható
frame_duration = display_time_per_image_seconds * fps  # Összes frame egy képhez
max_width = 1920  # Maximum szélesség (Full HD)
max_height = 1080  # Maximum magasság (Full HD)
white_threshold = 240  # A "fehér" érték threshold-je (0-255 skála)
white_ratio_threshold = 0.99  # Nagyon szigorú fehér arány (99%-os fehér)
min_width = 50  # Minimum szélesség érvényes képként
min_height = 50  # Minimum magasság érvényes képként

# Ellenőrizzük, hogy létezik-e a kimeneti mappa, ha nem, akkor hozzuk létre
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"A kimeneti mappa ({output_folder}) sikeresen létrehozva.")

def resize_image(image, max_width, max_height):
    """Kép méretének csökkentése, ha túl nagy."""
    if image is None or image.size == 0:
        print("HIBA: Üres vagy érvénytelen kép, kihagyva.")
        return None

    height, width = image.shape[:2]

    # Ellenőrizzük, hogy a kép mérete érvényes-e
    if width <= 0 or height <= 0:
        print(f"HIBA: Érvénytelen képméret: {width}x{height}. Kihagyva.")
        return None

    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        new_size = (int(width * scale), int(height * scale))
        print(f"Kép átméretezve: {width}x{height} -> {new_size[0]}x{new_size[1]}")
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def center_image(image, target_width, target_height, background_color=(255, 255, 255)):
    """Kép középre igazítása és kitöltése háttérszínnel."""
    if image is None or image.size == 0:
        print("HIBA: Üres vagy érvénytelen kép, nem lehet középre igazítani. Kihagyva.")
        return None

    height, width = image.shape[:2]
    
    # Ellenőrizzük, hogy a kép mérete érvényes-e
    if width <= 0 or height <= 0:
        print(f"HIBA: Érvénytelen képméret: {width}x{height}. Kihagyva.")
        return None

    padded_image = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)
    y_offset = (target_height - height) // 2
    x_offset = (target_width - width) // 2
    
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
    print(f"Kép középre igazítva: Méret={width}x{height}, Pályázati méret={target_width}x{target_height}")
    return padded_image

def is_white_row(gray_row, threshold=white_threshold, ratio_threshold=white_ratio_threshold):
    """
    Meghatározza, hogy egy adott sor több mint 'ratio_threshold' arányban fehér-e.
    """
    white_pixels = np.sum(gray_row > threshold)
    total_pixels = gray_row.size
    white_ratio = white_pixels / total_pixels
    return white_ratio > ratio_threshold

def extract_images_from_single_canvas(canvas):
    """
    Kinyeri az összes érvényes képet egyetlen nagy képből horizontális skennelés alapján.
    """
    all_images = []
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    start_y = None
    end_y = None

    print("Érvényes képek kinyerése a teljes canvas-ról...")

    # Találjuk meg az első nem fehér sort
    for y in range(height):
        if not is_white_row(gray[y, :]):
            if start_y is None:
                start_y = y  # Kezdési pozíció megtalálva
        else:
            if start_y is not None:
                end_y = y - 1  # Végső pozíció megtalálva
                extracted_region = canvas[start_y:end_y + 1, :]
                
                # Ellenőrizzük, hogy a kinyert régió mérete érvényes-e
                if extracted_region.shape[0] >= min_height and extracted_region.shape[1] >= min_width:
                    print(f"Érvényes kép kinyerve: Start_y={start_y}, End_y={end_y}, Méret={extracted_region.shape[1]}x{extracted_region.shape[0]}")
                    all_images.append(extracted_region)
                else:
                    print(f"HIBA: Túl kicsi régió kihagyva: Start_y={start_y}, End_y={end_y}, Méret={extracted_region.shape[1]}x{extracted_region.shape[0]}")
                
                start_y = None  # Új kezdési pozíció keresése

    # Ha van még nem lezárult kép a végén
    if start_y is not None and start_y < height - 1:
        end_y = height - 1
        extracted_region = canvas[start_y:end_y + 1, :]
        
        # Ellenőrizzük, hogy a kinyert régió mérete érvényes-e
        if extracted_region.shape[0] >= min_height and extracted_region.shape[1] >= min_width:
            print(f"Érvényes kép kinyerve: Start_y={start_y}, End_y={end_y}, Méret={extracted_region.shape[1]}x{extracted_region.shape[0]}")
            all_images.append(extracted_region)
        else:
            print(f"HIBA: Túl kicsi régió kihagyva: Start_y={start_y}, End_y={end_y}, Méret={extracted_region.shape[1]}x{extracted_region.shape[0]}")

    return all_images

def merge_pdf_pages_to_canvas(pdf_pages):
    """
    Egyesíti a PDF oldalakat egyetlen nagy képpé (canvas).
    """
    if not pdf_pages:
        print("Nincs feldolgozható oldal.")
        return None

    # Meghatározzuk a legnagyobb szélességet és összesítjük a magasságot
    max_width = max(page.shape[1] for page in pdf_pages)
    total_height = sum(page.shape[0] for page in pdf_pages)

    # Létrehozunk egy üres háttért a canvasnak
    canvas = np.full((total_height, max_width, 3), 255, dtype=np.uint8)

    # Rendezzük az oldalakat a canvasra
    current_y = 0
    for page in pdf_pages:
        height, width, _ = page.shape
        canvas[current_y:current_y + height, :width] = page
        current_y += height

    print(f"PDF oldalak egyetlen canvas-re rendezve. Méret: {max_width}x{total_height}")
    return canvas

def pdf_to_single_canvas(pdf_path):
    """Egy PDF fájl oldalait egyetlen nagy képpé konvertálja."""
    try:
        print(f"PDF konvertálása egyetlen képpé: {pdf_path}")
        images = convert_from_path(pdf_path)  # PDF oldalak konvertálása képekké
        pages = [np.array(img)[:, :, ::-1] for img in images]  # RGB -> BGR (OpenCV formátum)
        return merge_pdf_pages_to_canvas(pages)
    except Exception as e:
        print(f"Hiba történt a PDF feldolgozásakor ({pdf_path}): {e}")
        return None

def create_video_from_images(image_list, output_path, fps):
    """Képekből videót készít."""
    if not image_list:
        print(f"Üres képlistát kapott a videó generáláshoz: {output_path}")
        return

    resized_images = []
    for img_idx, img in enumerate(image_list):
        print(f"\nKép feldolgozása: {img_idx + 1}/{len(image_list)}")

        # Ellenőrizzük, hogy a kép nem üres és érvényes méretű
        if img is None or img.size == 0:
            print("HIBA: Üres vagy érvénytelen kép, kihagyva.")
            continue

        # Átméretezés és középre igazítás
        resized_img = resize_image(img, max_width, max_height)
        if resized_img is None:
            print("HIBA: A kép átméretezése sikertelen, kihagyva.")
            continue

        centered_img = center_image(resized_img, max_width, max_height)
        if centered_img is None:
            print("HIBA: A kép középre igazítása sikertelen, kihagyva.")
            continue

        resized_images.append(centered_img)

    if not resized_images:
        print(f"Nincs megjelenítendő kép a videóhoz: {output_path}")
        return

    # Videó író inicializálása
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 kodek
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (max_width, max_height))
    print(f"Videó írása elkezdve: {output_path}")

    # Minden képet hozzáadunk a videóhoz
    for img_idx, img in enumerate(resized_images):
        for frame_idx in range(frame_duration):  # Minden kép 5 másodpercig jelenik meg
            video_writer.write(img)
            if frame_idx % 30 == 0:  # Minden másodpercben frissítjük a státuszt
                print(f"Kép {img_idx + 1} framet írás: {frame_idx + 1}/{frame_duration}")

    video_writer.release()
    print(f"Videó sikeresen létrehozva: {output_path}")

def process_pdfs(input_folder, output_folder):
    """Feldolgozza az összes PDF-t a megadott mappából."""
    pdf_files = [f for f in os.listdir(input_folder) if f.startswith("Chapter") and f.endswith(".pdf")]
    pdf_files.sort()  # Rendezzük a fájlokat neve szerint

    all_videos = []  # Az összes kis videó elérési útja

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        chapter_number = pdf_file.split(" ")[1].split(".")[0]  # Ch 001.pdf -> 001
        output_video_name = f"Solo Leveling Chapter {chapter_number}.mp4"
        output_video_path = os.path.join(output_folder, output_video_name)

        print(f"\nFeldolgozás: {pdf_file} -> {output_video_name}")

        try:
            # PDF konvertálása egyetlen canvas-ra
            canvas = pdf_to_single_canvas(pdf_path)
            if canvas is None:
                print(f"Nincs feldolgozható tartalom a PDF-ben: {pdf_file}")
                continue

            # Képek kinyerése a canvas-ról
            extracted_images = extract_images_from_single_canvas(canvas)
            if not extracted_images:
                print(f"Nincs érvényes kép a PDF-ben: {pdf_file}")
                continue

            # Képekből videó generálása
            create_video_from_images(extracted_images, output_video_path, fps)
            all_videos.append(output_video_path)  # Hozzáadjuk az összes videóhoz
        except Exception as e:
            print(f"Hiba történt a feldolgozás során ({pdf_file}): {e}")

    # Ha minden kis videó kész, akkor egy nagy videót csinálunk
    #if all_videos:
    #    print("\nÖsszes kis videó feldolgozva. Nagy videó előállítása...")
    #    create_combined_video(all_videos, os.path.join(output_folder, "Solo Leveling - all.mp4"), fps)
    #else:
    #    print("Nincs feldolgozható videó, nagy videó nem hozható létre.")

def create_combined_video(video_list, output_path, fps):
    """Több videót egy nagy videóba fűz össze."""
    if not video_list:
        print("Nincs egyedi videó, amit összefűznünk kell.")
        return

    # Videó olvasó inicializálása az első videón
    cap = cv2.VideoCapture(video_list[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Nagy videó írása elkezdve: {output_path}")

    # Minden videót feldolgozzuk
    for video_idx, video in enumerate(video_list):
        print(f"\nVideó fűzés: {video} ({video_idx + 1}/{len(video_list)})")
        cap = cv2.VideoCapture(video)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            video_writer.write(frame)
            frame_count += 1
            if frame_count % 900 == 0:  # Minden 30 másodpercenként frissítjük a státuszt
                print(f"Frame-ek fűzve: {frame_count}")
        cap.release()

    video_writer.release()
    print(f"Nagy videó sikeresen létrehozva: {output_path}")

if __name__ == "__main__":
    print("Program indítása...")
    process_pdfs(input_folder, output_folder)
    print("Program befejeződött.")
