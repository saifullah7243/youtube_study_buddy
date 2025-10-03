@echo off
REM === Make folders ===

mkdir FrontEnd
mkdir BackEnd\src

REM === Create Folder ===
type nul > BackEnd\src\__init__.py
type nul > BackEnd\src\helper.py
type nul > BackEnd\src\utils.py
type nul > BackEnd\main.py
type nul > FrontEnd\__init__.py
type nul > FrontEnd\app.py
type nul > .env
type nul > setup.py
type nul > requirements.txt

echo Directory and files created successfully!.


