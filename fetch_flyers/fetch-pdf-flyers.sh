# This scripts downloads all documents from the Zenya webshare and puts them into a subfolder named 'flyers'.
# Author: Chris Josten (c.j.l.josten@student.utwente.nl)

################################################################################
# Config                                                                       #
################################################################################

# Url to the webshare, without trailing slash and no trailing 'DocumentsList.aspx'
# Looks like "https://webshare.zenya.work/****************"
webshare_url="https://webshare.zenya.work/sz79x7yjo2n75k5o"
# Location of cookie file, in Netscape format (http://fileformats.archiveteam.org/index.php?title=Netscape_cookies.txt)
cookie_file="cookies.txt"
# Location of CSV file containing the ID, file type and file name of each flyer
flyer_data_file="flyer-ids.csv"
# Path to directory where the flyers should be stored.
output_dir="flyers"

################################################################################
# Start of script                                                              #
################################################################################
err=0 # Keep track of errors that may have occured
IFS=";" # Set the input separator for the file
mkdir -p "$output_dir"
{
	# Skip first line
	read
	while read -r id filetype filename tail; do
		filename=`echo "$filename" | tr '/' '_'`
		ext="pdf"
		if [[ -f "$output_dir/$filename.$ext" ]]; then
			echo "$id ($filename.$ext) already exists"
		else
			wget --load-cookies "$cookie_file" -O "$output_dir/$filename.$ext" "${webshare_url}/DocumentResource/${id}/Document.html?showinlinepdf=1" || err=1;
			echo "$output_dir/$filename.$ext" "${webshare_url}/DocumentResource/${id}/Document.html?showinlinepdf=1"
		fi
	done 
} <$flyer_data_file

if [[ $err -ne 0 ]]; then
	echo "Errors detected: send message to Chris"
else
	echo "Donwloading done, no errors detected!"
fi
