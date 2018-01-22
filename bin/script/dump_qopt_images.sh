python src/resize.py
for file in `cd images/ && ls`; do
    bin/Release/guetzli resized_images/$file opt_images/$file qopt_images/$file
    echo "$file done"
done
python src/img2imgDataset.py