# test_grpc.py (at repo root)

import sys
sys.path.insert(0, 'src')

import grpc
import rag_service_pb2
import rag_service_pb2_grpc


def test(message: str = "What courses are in the first semester of Data Science?"):
    channel = grpc.insecure_channel('localhost:50051')
    stub = rag_service_pb2_grpc.MessageServiceStub(channel)

    request = rag_service_pb2.NewMessageRequest(
        conversation_id=1,
        user_message=message,
    )

    print(f"Sending: {message}\n")
    print("=" * 50)
    print("RESPONSE:")
    print("=" * 50)

    full_text = []

    for response in stub.GenerateReply(request):
        if response.HasField('token'):
            print(response.token.text, end='', flush=True)
            full_text.append(response.token.text)
        elif response.HasField('completion'):
            print(f"\n\n[DONE]")
        elif response.HasField('error'):
            print(f"\n[ERROR] {response.error.message}")

    channel.close()


if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "What courses are in the first semester of Data Science?"
    test(message)