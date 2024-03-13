# Flyer download script
This scripts downloads all flyers from the Zenya webshare and puts them into a subfolder named 'flyers'.
## Requirements
- Bash
- wget

## Instructions

1. Visit the webshare in your browser and export the cookies to a file named `cookies.txt` in the Netscape cookies format. Place it in the same folder as this README. I used this extension for my browser: https://github.com/rotemdan/ExportCookies. Also https://chromewebstore.google.com/detail/get-cookiestxt-clean/ahmnmhfbokciafffnknlekllgcnafnie can be used.
2. Open the developer console in your browser and evaluate the following code:
   ```js
   let csv = "id;filetype;filename\n" + window.objListview._objItems.map(x => x.ID + ";" + x.ColumnValues.type.ListValues[0].ID + ";" + x.ColumnValues.title.TextValue).join("\n")
   let file = new File([csv], "flyer-ids.csv", {type: "text/csv;charset=utf-8;header=present"});
   let exportUrl = URL.createObjectURL(file);
   window.location.assign(exportUrl);
   URL.revokeObjectURL(exportUrl);
   ```
3. Move the downloaded file to the same folder as this README.
4. Modify `fetch-flyers.sh` and set the `webshare_url` variable to the right value (see comment above the variable for instructions)
5. Run `fetch-flyers.sh`
