for file in `cd images/ && ls`; do
    bin/Release/guetzli images/$file opt_images/$file qopt_images/$file
    echo "$file done"
done

python csv2train_data.py
