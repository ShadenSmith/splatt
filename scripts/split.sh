#!/bin/bash

if [ "$#" -eq 0 ]; then
  echo "usage: shift <tensor.tns>";
  exit 1;
fi

echo "Splitting $1 80/10/10."

shuf $1 > tmp.tns;
split -a 1 -n l/10 -d tmp.tns;

cat x{0..7} > train.tns;
rm x{0..7};

mv x8 val.tns;
mv x9 test.tns;

rm tmp.tns;

