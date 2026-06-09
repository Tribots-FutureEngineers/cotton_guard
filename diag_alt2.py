import os, sys
sys.path.insert(0, r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Check what images are actually in the aug_Alternaria_Leaf training folder
alt_train = r'E:\DL AGRICULTURE\aug_Alternaria_Leaf'
print(f'=== aug_Alternaria_Leaf folder ===')
if os.path.exists(alt_train):
    files = [f for f in os.listdir(alt_train) if os.path.isfile(os.path.join(alt_train, f))]
    print(f'Total files: {len(files)}')
    from PIL import Image
    sample = files[:5]
    for f in sample:
        try:
            img = Image.open(os.path.join(alt_train, f))
            print(f'  {f[:60]:60s} -> {img.size} {img.mode}')
        except Exception as e:
            print(f'  {f}: ERROR {e}')
else:
    print('FOLDER NOT FOUND')

print()
print(f'=== Alternaria Leaf Spot Cotton (extra) ===')
alt_extra = r'C:\Users\DRAGON CENTER\Downloads\CottonGuard_Offline_Deploy\Alternaria Leaf Spot Cotton'
if os.path.exists(alt_extra):
    files2 = [f for f in os.listdir(alt_extra) if os.path.isfile(os.path.join(alt_extra, f))]
    print(f'Total files: {len(files2)}')
    sample2 = files2[:5]
    for f in sample2:
        try:
            img = Image.open(os.path.join(alt_extra, f))
            print(f'  {f[:60]:60s} -> {img.size} {img.mode}')
        except Exception as e:
            print(f'  {f}: ERROR {e}')
else:
    print('FOLDER NOT FOUND')
