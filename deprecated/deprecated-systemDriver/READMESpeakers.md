requirements file just has every dependency currently on the pi (kept in case it gets reflashed)
- install with `pip3 install -r requirements.txt`
final script is "generative_audio.py" (run `python3 generative_audio.py`)

usb_to_audio.py is a simple test script to check that audio can be played to begin with

Import speak() function via:
`from speakers import speak`

Examplse of using peak():
`speak("Sample text")` - plays "Sample text" in google text to speech on both speakers, at 30% vol
`speak("Sample audio!", 1)` plays "Sample audio!" on the left speaker at 30% vol
`speak("Sample sound!!", 2, 60)` plays "Sample sound!!" on right speaker at 60% vol
`speak("Sample text", vol=40)` plays sound at 40% vol on both speakers
`speak("Sample text", 3, 20)` plays 20% volume in both speakers