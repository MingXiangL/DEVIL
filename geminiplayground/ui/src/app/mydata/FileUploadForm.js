'use client';
import {forwardRef, useEffect, useImperativeHandle, useState} from "react";
import {useForm} from "react-hook-form";
import {zodResolver} from "@hookform/resolvers/zod";
import {Dialog, DialogContent, DialogHeader, DialogTitle} from "@/components/ui/dialog";
import {Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage} from "@/components/ui/form";
import {Input} from "@/components/ui/input";
import {Button} from "@/components/ui/button";
import {z} from "zod";
import {useMutation, useQueryClient} from "@tanstack/react-query";
import {axiosInstance} from "@/app/axios";
import {Loader2} from "lucide-react"
import {useToast} from "@/components/ui/use-toast"
import {ToastAction} from "@/components/ui/toast";


const MAX_FILE_SIZE = 1024 * 1024 * 70; // 70MB
const ALLOWED_FILE_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/jpg",
    "video/mp4",
    'audio/mpeg',
    'audio/wav',
    'audio/mp3',
    "application/pdf",
];
// Form Schema Validation
const fileUploadFormSchema = z.object({
    file: z.any()
        .refine((file) => {
            return file !== null;
        }, {
            message: "File is required"
        })
        .refine((file) => {
            return file?.size <= MAX_FILE_SIZE && ALLOWED_FILE_TYPES.includes(file?.type);
        }, {
            message: `The file must be less than ${MAX_FILE_SIZE / 1024 / 1024}MB and of type ${ALLOWED_FILE_TYPES.join(", ")}`,
        })
});
const FileUploadForm = forwardRef(function FileUploadForm(props, ref) {
    const [open, setOpen] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const queryClient = useQueryClient();
    const {toast} = useToast();

    const form = useForm({
        resolver: zodResolver(fileUploadFormSchema),
        defaultValues: {
            file: null
        }
    });
    const mutate = useMutation({
        mutationFn: async (data) => {
            return axiosInstance.post("/uploadFile", data, {
                headers: {
                    "Content-Type": "multipart/form-data"
                }
            });
        }
    });

    const onSubmit = async (formValues, evt) => {
        try {
            evt.preventDefault();
            setIsSubmitting(true);
            const formData = new FormData();
            formData.append("file", formValues.file);
            const result = await mutate.mutateAsync(formData);
            await queryClient.refetchQueries(["parts"]);

        } catch (error) {
            console.error("Error uploading file", error);
            const errorMessage = error.response?.data?.detail || error.message;
            toast({
                variant: "destructive",
                title: "Uh oh! Something went wrong.",
                description: `An error occurred while uploading the file. ${errorMessage}`,
                action: <ToastAction altText="Try again">Close</ToastAction>,
            })
        } finally {
            form.reset();
            setOpen(false);
            setIsSubmitting(false);
        }
    }


    useImperativeHandle(ref, () => ({
        open: () => setOpen(true),
        close: () => setOpen(false),
        reset: () => form.reset(),
    }), []);

    // const uploadedFile = form.watch("file");
    // //console.log("uploading image", image);

    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle>Files</DialogTitle>
                    <Form {...form}>
                        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                            <FormField
                                control={form.control}
                                name="file"
                                render={({field: {value, onChange, ...fieldProps}}) => (
                                    <FormItem>
                                        <FormLabel>File</FormLabel>
                                        <FormControl>
                                            <Input type="file"
                                                   {...fieldProps}
                                                   onChange={(event) => {
                                                       const file = event.target.files[0];
                                                       //form.setValue("imageFile", file);
                                                       onChange(file);
                                                   }}/>
                                        </FormControl>
                                        <FormDescription>
                                            Upload a file of maximum size {MAX_FILE_SIZE / 1024 / 1024}MB. Allowed file
                                            types: {ALLOWED_FILE_TYPES.join(", ")}
                                        </FormDescription>
                                        <FormMessage/>
                                    </FormItem>
                                )}
                            />
                            {isSubmitting ? (
                                <Button type="submit" disabled>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin"/>
                                    Please wait...
                                </Button>
                            ) : (
                                <Button type="submit">Submit</Button>
                            )}
                        </form>
                    </Form>
                </DialogHeader>
            </DialogContent>
        </Dialog>
    );
});

export default FileUploadForm;