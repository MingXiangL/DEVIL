# Navigate to the ui directory and run npm build
Push-Location ./ui
npm run build
Pop-Location

# After building, navigate to the out directory and copy the files to the target directory
Push-Location ./ui/out
Copy-Item -Path * -Destination ../../src/geminiplayground/web/static -Recurse -Force
Pop-Location
