#!/bin/bash
echo "Aizip Smart TV API Function List:"
cat <<EOF 
def search_videos(query, app):
    """
    Searches videos matching a query on the specified app.
    Parameters:
    - query (str): Search query.
    - app (str): Name of the app.
    Returns:
    - list[str]: A list of strings, each string includes video names and URLs.
    """
def set_input_source(source):
    """
    Set the TV input source.
    Parameters:
    - source (enum): TV input source. Select from “TV”, “HDMI”
    Returns:
    """
def pause_video():
    """
    Pause the video.
    Parameters:
    Returns:
    """
def resume_video():
    """
    Resume the video.
    Parameters:
    Returns:
    """
def volume_up():
    """
    Increase the volume.
    Parameters:
    Returns:
    """
def volume_down():
    """
    Lower the volume.
    Parameters:
    Returns:
    """
EOF
echo "Enter a query (or type 'exit' to quit): "
# Infinite loop to continuously receive arguments
while true; do
  # Prompt the user for an argument
  read input
  
  # Check if the user wants to exit
  if [ "$input" == "exit" ]; then
    echo "Exiting the script."
    break
  fi
  
  # Run the function with the provided argument
  ./aizip-agent -m /home/root/aizip_smartTV_demo_ggml-model-Q4_0_4_4_wopt.gguf -p "$input" --temp 0 -c 512 2>/home/root/error.txt
  #process_argument $input
done