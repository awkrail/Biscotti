for file in `cd images/ && ls`; do
    bin/Release/qopt_guetzli --verbose images/$file qopt_images/$file
done
