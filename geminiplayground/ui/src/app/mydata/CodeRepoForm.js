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
import {Loader2} from "lucide-react";
import {ToastAction} from "@/components/ui/toast";
import {useToast} from "@/components/ui/use-toast";


// Form Schema Validation
const codeRepoFormSchema = z.object({
    repoPath: z.string().min(1),
    repoBranch: z.string().min(1)
});
const CodeRepoForm = forwardRef(function ImagesFileUpload(props, ref) {
    const [open, setOpen] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const {toast} = useToast();
    const queryClient = useQueryClient();

    const form = useForm({
        resolver: zodResolver(codeRepoFormSchema),
        defaultValues: {
            repoPath: null,
            repoBranch: "main"
        }
    });
    const mutate = useMutation({
        mutationFn: async (data) => {
            return axiosInstance.post("/uploadRepo", data, {
                headers: {
                    "Content-Type": "application/json"
                }
            });
        }
    });

    const onSubmit = async (formValues, evt) => {
        try {
            console.log("Form Values", formValues);
            evt.preventDefault();
            setIsSubmitting(true);
            const result = await mutate.mutateAsync(formValues);
            console.log("Result", result);
            await queryClient.refetchQueries(["parts"]);

        } catch (error) {
            console.error("Error uploading file", error);
            const errorMessage = error.response?.data?.detail || error.message;
            toast({
                variant: "destructive",
                title: "Uh oh! Something went wrong.",
                description: `An error occurred creating the repository: ${errorMessage}`,
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


    return (
        <Dialog open={open} onOpenChange={setOpen}>
            <DialogContent>
                <DialogHeader>
                    <DialogTitle>Github Repository</DialogTitle>
                    <Form {...form}>
                        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                            <FormField
                                control={form.control}
                                name="repoPath"
                                render={({field}) => (
                                    <FormItem>
                                        <FormLabel>Repo Path</FormLabel>
                                        <FormControl>
                                            <Input type={"text"} {...field}/>
                                        </FormControl>
                                        <FormDescription>
                                            Enter the folder path(In your local machine) or the url of the repository
                                        </FormDescription>
                                        <FormMessage/>
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="repoBranch"
                                render={({field}) => (
                                    <FormItem>
                                        <FormLabel>Branch</FormLabel>
                                        <FormControl>
                                            <Input type={"text"} {...field}/>
                                        </FormControl>
                                        <FormDescription>
                                            Enter the branch name of the repository
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

export default CodeRepoForm;