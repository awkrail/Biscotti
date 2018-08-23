cd guetzli_dumper && make
cd .. && mv guetzli_dumper/bin/Release/guetzli train_bin/Release/guetzli_dumper
python src/mkdir.py
python src/resize.py
for file in `cd images/ && ls`; do
    train_bin/Release/guetzli_dumper resized_images/$file opt_images/$file qopt_images/$file
    echo "$file done"
done
python src/img2imgDataset.py