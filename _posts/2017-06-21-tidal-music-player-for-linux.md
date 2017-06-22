---
layout: post
title: "Tidal music player for Linux"
date: 2017-06-21
header: true
footer: true
comments: false
tags: 
---

I found an awesome package called [tidal-music-linux](https://github.com/Bunkerbewohner/tidal-music-linux) that wraps a Chromium web player in an Electron shell for a standalone Tidal player like on Windows. It's written by [Mathias Kahl](https://github.com/Bunkerbewohner).

<img src='/images/tidal-music-player/screenshot.png' style='width: 100%; object-fit: contain'/>

It should work in either Ubuntu or Arch. A major drawback is that I haven't gotten HiFi playback to work. It says HiFi is only allowed in Chrome. Maybe wrapping a Chrome player in Electron rather than a Chromium player would solve this. Regardless, here's how to install if you're okay with 320 kbps playback in a beautiful standalone application.

## Arch Instructions

* Install [Pepper Flash Player](https://aur.archlinux.org/packages/pepper-flash/): 
	* `yaourt -S pepper-flash`
* Install [Adobe Flash Player](https://get.adobe.com/flashplayer/):
	* Download the .tar.gz bundle from the link above
	* Move the .tar.gz bundle to `usr/lib/adobe-flashplugin`
		* `mkdir /usr/lib/adobe-flashplugin`
		* `sudo mv ~/Downloads/flash_player_ppapi_linux.x86_64.tar.gz /usr/lib/adobe-flashplugin`
* Install [Node Package Manager (NPM)](https://www.archlinux.org/packages/community/any/npm/)
	* `sudo pacman -S npm`
* Download
	* `git clone https://github.com/Bunkerbewohner/tidal-music-linux`
* Change name
	* Inside package.json the name "Tidal music - Linux" creates an install error, so change it to something like "tidal-music-linux"
		* `gedit tidal-music-linux/package.json`
		* Change the second line to "name": "tidal-music-linux"
* Install
	* `cd tidal-music-linux`
	* `npm install`
* Run
	* `npm start`

