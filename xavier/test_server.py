from paho.mqtt import client as mqtt_client

broker = '192.168.100.2'
port = 1883
topic = "ack"
topic_sub = "audios"
client_id = 'xavier'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Successfully connected to MQTT broker")
        else:
            print("Failed to connect, return code %d", rc)
    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        f = open('receive.mp3', 'wb')
        f.write(msg.payload)
        f.close()
        print ('Audio received')

    client.subscribe(topic_sub)
    client.on_message = on_message

def main():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()
    
if __name__ == '__main__':
    main()
