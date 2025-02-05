# delete file detection_log.txt if exists

if (Test-Path -Path .\detection_log.txt) {
    Remove-Item -Path .\detection_log.txt
}

if (Test-Path -Path .\detection_results.json) {
    Remove-Item -Path .\detection_results.json
}

# delete all folder and files in frames folder
Remove-Item -Path .\frames\* -Recurse