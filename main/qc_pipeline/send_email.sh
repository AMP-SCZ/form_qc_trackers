#!/bin/bash


message="$1"

recipient="$2"
subject="$3"



function send_email() {
    recipient="$1"
    subject="$2"
    message="$3"
    echo "$message" | mail -s "$subject" "$recipient"
}
send_email "$recipient" "$subject" "$message"
