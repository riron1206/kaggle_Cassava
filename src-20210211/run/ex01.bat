@echo off
set CASE=01
set DATA=v1

if "%2" == "" (
    if "%1" == "" (
        set KS=0
        set KE=4
    ) else (
        set KS=%1
        set KE=%1
    )
) else (
    set KS=%1
    set KE=%2
)
for /l %%i in (%KS%,1,%KE%) do (
    call :RUN %%i
)
exit /b

:RUN
echo --- FOLD=%1 ---
python train.py ^
--train_csv=csvs/%DATA%/k%1_train.csv ^
--valid_csv=csvs/%DATA%/k%1_valid.csv ^
--image_dir=data/images ^
--output_dir=outputs/case%CASE%/k%1/ ^
--epoch=25 ^
--scheduler=sc1:1000 ^
--image_size=512 ^
--train_augs=ex2 ^
--undersample=4000

exit /b