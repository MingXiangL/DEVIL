"use client";
import ChatMessagesBox from "@/components/Chat/ChatMessagesBox";
import ChatInputBox from "@/components/Chat/ChatInputBox";
import {useMutation, useQuery, useQueryClient} from "@tanstack/react-query";
import {axiosInstance} from "@/app/axios";
import moment from "moment";
import ChatSettings from "@/components/ChatSettings";
import {useEffect, useRef} from "react";
import useWebSocket, {ReadyState} from "react-use-websocket";

function removeBrackets(str) {
    return str.replace(/[\[\](){}<>]/g, '');
}

const WS_URL = `ws://${process.env.NEXT_PUBLIC_API_BASE_URL}/ws`;


export default function Chat() {

    const queryClient = useQueryClient();
    const settingFormRef = useRef(null);
    const {sendJsonMessage, lastJsonMessage, readyState} = useWebSocket(
        WS_URL,
        {
            share: false,
            shouldReconnect: () => false,
        },
    )
    const {data: selectedModel} = useQuery({
        queryKey: ["selectedModel"],
        queryFn: async () => {
            return queryClient.getQueryData(["selectedModel"]) || null;
        }
    });

    useEffect(() => {
        console.log("Connection state changed")
        if (readyState === ReadyState.OPEN) {
            sendJsonMessage({
                event: "subscribe",
                data: {
                    channel: "messages"
                }
            })
        }
    }, [sendJsonMessage, readyState])


    const apiEndpoint = "/tags";
    const {data: tagsData, isLoading, refetch} = useQuery({
        queryKey: [apiEndpoint],
        queryFn: async () => {
            const res = await axiosInstance.get(apiEndpoint);
            return res.data;
        }
    });

    const {data: messages} = useQuery({
        queryKey: ["messages"],
        queryFn: async () => {
            return queryClient.getQueryData(["messages"]) || [];
        }
    });


    const sendMessage = useMutation({
        mutationFn: async (messageRequest) => {
            sendJsonMessage({
                event: "generate_response",
                data: messageRequest
            });
        }
    });

    // Run when a new WebSocket message is received (lastJsonMessage)
    useEffect(() => {
        const {event, data} = lastJsonMessage || {};
        if (event === "response_completed") {
            updateLastMessage.mutate({
                loading: false
            });
        } else if (event === "response_chunk") {
            const lastMessage = getLastMessage();
            if (lastMessage) {
                const {content} = lastMessage;
                const updatedContent = content + data;
                updateLastMessage.mutate({
                    content: updatedContent,
                    loading: false
                });
            }
        } else if (event === "response_error") {
            updateLastMessage.mutate({
                content: "Ups! Something went wrong. I cannot generate a response at the moment",
                loading: false,
                error: data
            });
        }
    }, [lastJsonMessage])

    useEffect(() => {
        if (selectedModel) {
            sendJsonMessage({
                event: "set_model",
                data: {
                    model: selectedModel
                }
            });
        }
    }, [selectedModel])


    const addMessage = useMutation({
        mutationFn: async (message) => {
            queryClient.setQueryData(["messages"], (prevMessages) => {
                return [...prevMessages, message];
            });
        }
    });

    const updateMessage = useMutation({
        mutationFn: async (message) => {
            queryClient.setQueryData(["messages"], (prevMessages) => {
                return prevMessages.map((msg) => {
                    if (msg.timestamp === message.timestamp) {
                        return message;
                    }
                    return msg;
                });
            });
        }
    });

    const updateLastMessage = useMutation({
        mutationFn: async (message) => {
            queryClient.setQueryData(["messages"], (prevMessages) => {
                const updatedLastMessage = {
                    ...prevMessages[prevMessages.length - 1],
                    ...message
                };
                return [...prevMessages.slice(0, -1), updatedLastMessage];
            });
        }
    });

    const getLastMessage = () => {
        return queryClient.getQueryData(["messages"])[queryClient.getQueryData(["messages"]).length - 1];
    }

    function addMessageToChat(role, content, isLoading, rawMessage, error = null) {
        const timestamp = new Date().toISOString();
        const momentFormatted = moment().format('MMMM Do YYYY, h:mm:ss a');
        const message = {
            role,
            content,
            timestamp,
            rawMessage,
            loading: isLoading,
            moment: momentFormatted
        };
        if (error) {
            message.error = error;
        }
        addMessage.mutate(message);
    }

    // Function to handle the generation of the response and update the UI accordingly
    async function handleResponseGeneration(message) {
        if (!message) {
            return;
        }
        try {

            const settingsForm = settingFormRef.current;
            const settingsValid = await settingsForm.validateSettings();
            if (!settingsValid) {
                return;
            }
            const model = queryClient.getQueryData(["selectedModel"]);
            let modelSettings = settingFormRef.current.getSettings();
            modelSettings = {
                temperature: parseFloat(modelSettings.temperature),
                top_k: parseInt(modelSettings.top_k),
                top_p: parseFloat(modelSettings.top_p),
            }

            const messageRequest = {
                model,
                message,
                settings: modelSettings
            };

            // Add user message to chat
            addMessageToChat("user", removeBrackets(message), false, message);

            // Add initial model message to chat
            addMessageToChat("model", "", true, message);

            // start the generation process
            await sendMessage.mutateAsync(messageRequest);

        } catch (e) {
            console.log("error in generating response", e);
            updateLastMessage.mutate({
                content: "Ups! Something went wrong. I cannot generate a response at the moment",
                loading: false,
                error: e
            });
        }
    }

    const onMessageSend = async (message, evt) => {
        // add user message to chat
        await handleResponseGeneration(message);
    }

    const clearChatQueueHandler = async () => {
        queryClient.setQueryData(["messages"], []);
        sendJsonMessage({
            event: "clear_queue",
            data: {}
        });
    }
    return (
        <div className="flex flex-col lg:flex-row gap-3">
            <div className="w-full">
                <div
                    className="flex flex-col  rounded-xl bg-muted/50 p-1 border-2 h-screen lg:h-[calc(100vh-90px)] w-full m-0">
                    <div className="w-full h-dvh overflow-auto">
                        <ChatMessagesBox messages={messages} onClearChatQueue={clearChatQueueHandler}/>
                    </div>
                    <div className="w-full">
                        <ChatInputBox inputTags={tagsData} onMessageSend={onMessageSend} tagValueAccessor="name"/>
                    </div>
                </div>
            </div>
            <div>
                <ChatSettings ref={settingFormRef}/>
            </div>
        </div>
    )
}