import os

import cv2
import tensorflow as tf

from keras.src.utils import image_dataset_from_directory


def extract_frames_from_video(video_path, output_folder, class_name, frame_rate=30):
    """
    Extrai frames de um vídeo e salva no diretório de saída com nome baseado na classe.

    Args:
    - video_path (str): Caminho do vídeo.
    - output_folder (str): Diretório onde os frames serão salvos.
    - class_name (str): Nome da classe para nomear os arquivos de imagem.
    - frame_rate (int): Taxa de frames por segundo para extrair.
    """
    # Abrir o vídeo
    cap = cv2.VideoCapture(video_path)

    # Obter o número total de frames do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Contador para nome sequencial dos arquivos de imagem
    frame_count = 0

    # Ler e salvar frames
    for i in range(0, total_frames, frame_rate):
        ret, frame = cap.read()
        if not ret:
            break

        # Nome do arquivo com base na classe e número sequencial
        frame_filename = os.path.join(output_folder, f"{class_name}_frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Incrementar o contador de frames
        frame_count += 1

    # Liberar o vídeo
    cap.release()


def extract_frames_from_videos_in_directory(path_video, output_root_dir, frame_rate):
    """
    Extrai frames de todos os vídeos dentro de subpastas em um diretório.

    Args:
    - path_video (str): Caminho do diretório contendo as subpastas de vídeos.
    - output_root_dir (str): Caminho do diretório onde os frames serão salvos.
    """

    for class_folder in os.listdir(path_video):
        class_path = os.path.join(path_video, class_folder)

        if os.path.isdir(class_path) and not os.path.exists(os.path.join(output_root_dir, class_folder)):
            print(f'Iniciando processamento da pasta "{class_folder}"...')

            # Criar diretório correspondente na pasta de saída
            output_class_folder = os.path.join(output_root_dir, class_folder)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)

            # Processar todos os vídeos dentro da pasta da classe
            for video_file in os.listdir(class_path):
                if video_file.endswith(".mp4") or video_file.endswith(".avi"):
                    print(f'Iniciando processamento do vídeo "{video_file}"...')

                    video_path = os.path.join(class_path, video_file)
                    extract_frames_from_video(video_path, output_class_folder, class_folder, frame_rate=frame_rate)
        else:
            print(f'O diretório "{class_folder}" já foi processado.')


def load_video_dataset(directory, sequence_length=10, img_height=128, img_width=128, batch_size=32):
    # Carrega o dataset de imagens individualmente
    dataset = image_dataset_from_directory(
        directory,
        image_size=(img_height, img_width),
        batch_size=1,  # Batch de 1, pois vamos juntar os frames manualmente
        label_mode='int',
        shuffle=False
    )

    # Função para transformar o dataset em sequência de frames
    def create_sequences(dataset, sequence_length):
        frames_sequence = []
        labels_sequence = []

        for image_batch, label_batch in dataset:
            frames_sequence.append(image_batch[0])  # Adiciona o frame individualmente
            if len(frames_sequence) == sequence_length:
                # Adiciona uma sequência completa ao dataset
                frames_sequence = tf.stack(frames_sequence)  # Stack para formar a sequência
                labels_sequence.append(label_batch[0])  # Usa o label do último frame da sequência

                # Limpa os frames para próxima sequência
                frames_sequence = tf.expand_dims(frames_sequence, axis=0)  # Adiciona dimensão batch
                yield frames_sequence, label_batch[0]  # Retorna um gerador para processar um item de cada vez

                frames_sequence = []  # Reinicia a sequência para a próxima janela

    # Criar o dataset final usando o gerador
    dataset = tf.data.Dataset.from_generator(
        lambda: create_sequences(dataset, sequence_length),
        output_signature=(
            tf.TensorSpec(shape=(sequence_length, img_height, img_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    # Aplica o batch final no dataset de sequências
    dataset = dataset.batch(batch_size)
    return dataset