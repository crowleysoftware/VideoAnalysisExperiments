# delete file detection_log.txt if exists

if (Test-Path -Path .\detection_log.txt) {
    Remove-Item -Path .\detection_log.txt
}

if (Test-Path -Path .\detection_results.json) {
    Remove-Item -Path .\detection_results.json
}

if (Test-Path -Path .\detection_sections.json) {
    Remove-Item -Path .\detection_sections.json
}

if (Test-Path -Path .\video_detection_log.txt) {
    Remove-Item -Path .\video_detection_log.txt
}

if (Test-Path -Path .\final_detection_log.txt) {
    Remove-Item -Path .\final_detection_log.txt
}

if (Test-Path -Path .\chosen.txt) {
    Remove-Item -Path .\chosen.txt
}


# delete all folder and files in frames folder
Remove-Item -Path .\frames\* -Recurse
Remove-Item -Path .\chosen_frames\* -Recurse
Remove-Item -Path .\full_frames\* -Recurse